import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from letta.constants import MCP_TOOL_TAG_NAME_PREFIX
from letta.errors import (
    AgentExportIdMappingError,
    AgentExportProcessingError,
    AgentFileExportError,
    AgentFileImportError,
    AgentNotFoundForExportError,
)
from letta.helpers.pinecone_utils import should_use_pinecone
from letta.helpers.tpuf_client import should_use_tpuf
from letta.log import get_logger
from letta.schemas.agent import AgentState, CreateAgent
from letta.schemas.agent_file import (
    AgentFileSchema,
    AgentSchema,
    BlockSchema,
    FileAgentSchema,
    FileSchema,
    GroupSchema,
    ImportResult,
    MCPServerSchema,
    MessageSchema,
    SourceSchema,
    ToolSchema,
)
from letta.schemas.block import Block
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import FileProcessingStatus, VectorDBProvider
from letta.schemas.file import FileMetadata
from letta.schemas.group import Group, GroupCreate
from letta.schemas.mcp import MCPServer
from letta.schemas.message import Message
from letta.schemas.source import Source
from letta.schemas.tool import Tool
from letta.schemas.user import User
from letta.services.agent_manager import AgentManager
from letta.services.block_manager import BlockManager
from letta.services.file_manager import FileManager
from letta.services.file_processor.embedder.openai_embedder import OpenAIEmbedder
from letta.services.file_processor.embedder.pinecone_embedder import PineconeEmbedder
from letta.services.file_processor.file_processor import FileProcessor
from letta.services.file_processor.parser.markitdown_parser import MarkitdownFileParser
from letta.services.file_processor.parser.mistral_parser import MistralFileParser
from letta.services.files_agents_manager import FileAgentManager
from letta.services.group_manager import GroupManager
from letta.services.mcp_manager import MCPManager
from letta.services.message_manager import MessageManager
from letta.services.source_manager import SourceManager
from letta.services.tool_manager import ToolManager
from letta.settings import settings
from letta.utils import get_latest_alembic_revision, safe_create_task

logger = get_logger(__name__)


class AgentSerializationManager:
    """
    Manages export and import of agent files between database and AgentFileSchema format.

    Handles:
    - ID mapping between database IDs and human-readable file IDs
    - Coordination across multiple entity managers
    - Transaction safety during imports
    - Referential integrity validation
    """

    def __init__(
        self,
        agent_manager: AgentManager,
        tool_manager: ToolManager,
        source_manager: SourceManager,
        block_manager: BlockManager,
        group_manager: GroupManager,
        mcp_manager: MCPManager,
        file_manager: FileManager,
        file_agent_manager: FileAgentManager,
        message_manager: MessageManager,
    ):
        self.agent_manager = agent_manager
        self.tool_manager = tool_manager
        self.source_manager = source_manager
        self.block_manager = block_manager
        self.group_manager = group_manager
        self.mcp_manager = mcp_manager
        self.file_manager = file_manager
        self.file_agent_manager = file_agent_manager
        self.message_manager = message_manager
        self.file_parser = MistralFileParser() if settings.mistral_api_key else MarkitdownFileParser()

        # ID mapping state for export
        self._db_to_file_ids: Dict[str, str] = {}

        # Counters for generating Stripe-style IDs
        self._id_counters: Dict[str, int] = {
            AgentSchema.__id_prefix__: 0,
            GroupSchema.__id_prefix__: 0,
            BlockSchema.__id_prefix__: 0,
            FileSchema.__id_prefix__: 0,
            SourceSchema.__id_prefix__: 0,
            ToolSchema.__id_prefix__: 0,
            MessageSchema.__id_prefix__: 0,
            FileAgentSchema.__id_prefix__: 0,
            MCPServerSchema.__id_prefix__: 0,
        }

    def _reset_state(self):
        """Reset internal state for a new operation"""
        self._db_to_file_ids.clear()
        for key in self._id_counters:
            self._id_counters[key] = 0

    def _generate_file_id(self, entity_type: str) -> str:
        """Generate a Stripe-style ID for the given entity type"""
        counter = self._id_counters[entity_type]
        file_id = f"{entity_type}-{counter}"
        self._id_counters[entity_type] += 1
        return file_id

    def _map_db_to_file_id(self, db_id: str, entity_type: str, allow_new: bool = True) -> str:
        """Map a database UUID to a file ID, creating if needed (export only)"""
        if db_id in self._db_to_file_ids:
            return self._db_to_file_ids[db_id]

        if not allow_new:
            raise AgentExportIdMappingError(db_id, entity_type)

        file_id = self._generate_file_id(entity_type)
        self._db_to_file_ids[db_id] = file_id
        return file_id

    def _extract_unique_tools(self, agent_states: List[AgentState]) -> List:
        """Extract unique tools across all agent states by ID"""
        all_tools = []
        for agent_state in agent_states:
            if agent_state.tools:
                all_tools.extend(agent_state.tools)

        unique_tools = {}
        for tool in all_tools:
            unique_tools[tool.id] = tool

        return sorted(unique_tools.values(), key=lambda x: x.name)

    def _extract_unique_blocks(self, agent_states: List[AgentState]) -> List:
        """Extract unique blocks across all agent states by ID"""
        all_blocks = []
        for agent_state in agent_states:
            if agent_state.memory and agent_state.memory.blocks:
                all_blocks.extend(agent_state.memory.blocks)

        unique_blocks = {}
        for block in all_blocks:
            unique_blocks[block.id] = block

        return sorted(unique_blocks.values(), key=lambda x: x.label)

    async def _extract_unique_sources_and_files_from_agents(
        self, agent_states: List[AgentState], actor: User, files_agents_cache: dict = None
    ) -> tuple[List[Source], List[FileMetadata]]:
        """Extract unique sources and files from agent states using bulk operations"""

        all_source_ids = set()
        all_file_ids = set()

        for agent_state in agent_states:
            files_agents = await self.file_agent_manager.list_files_for_agent(
                agent_id=agent_state.id,
                actor=actor,
                is_open_only=False,
                return_as_blocks=False,
                per_file_view_window_char_limit=agent_state.per_file_view_window_char_limit,
            )
            # cache the results for reuse during conversion
            if files_agents_cache is not None:
                files_agents_cache[agent_state.id] = files_agents

            for file_agent in files_agents:
                all_source_ids.add(file_agent.source_id)
                all_file_ids.add(file_agent.file_id)
        sources = await self.source_manager.get_sources_by_ids_async(list(all_source_ids), actor)
        files = await self.file_manager.get_files_by_ids_async(list(all_file_ids), actor, include_content=True)

        return sources, files

    async def _convert_agent_state_to_schema(self, agent_state: AgentState, actor: User, files_agents_cache: dict = None) -> AgentSchema:
        """Convert AgentState to AgentSchema with ID remapping"""

        agent_file_id = self._map_db_to_file_id(agent_state.id, AgentSchema.__id_prefix__)

        # use cached file-agent data if available, otherwise fetch
        if files_agents_cache is not None and agent_state.id in files_agents_cache:
            files_agents = files_agents_cache[agent_state.id]
        else:
            files_agents = await self.file_agent_manager.list_files_for_agent(
                agent_id=agent_state.id,
                actor=actor,
                is_open_only=False,
                return_as_blocks=False,
                per_file_view_window_char_limit=agent_state.per_file_view_window_char_limit,
            )
        agent_schema = await AgentSchema.from_agent_state(
            agent_state, message_manager=self.message_manager, files_agents=files_agents, actor=actor
        )
        agent_schema.id = agent_file_id

        # Ensure all in-context messages are present before ID remapping.
        # AgentSchema.from_agent_state fetches a limited slice (~50) and may exclude messages still
        # referenced by in_context_message_ids. Fetch any missing in-context messages by ID so remapping succeeds.
        existing_msg_ids = {m.id for m in (agent_schema.messages or [])}
        in_context_ids = agent_schema.in_context_message_ids or []
        missing_in_context_ids = [mid for mid in in_context_ids if mid not in existing_msg_ids]
        if missing_in_context_ids:
            missing_msgs = await self.message_manager.get_messages_by_ids_async(message_ids=missing_in_context_ids, actor=actor)
            fetched_ids = {m.id for m in missing_msgs}
            not_found = [mid for mid in missing_in_context_ids if mid not in fetched_ids]
            if not_found:
                # Surface a clear mapping error; handled upstream by the route/export wrapper.
                raise AgentExportIdMappingError(db_id=not_found[0], entity_type=MessageSchema.__id_prefix__)
            for msg in missing_msgs:
                agent_schema.messages.append(MessageSchema.from_message(msg))

        # wipe the values of tool_exec_environment_variables (they contain secrets)
        agent_secrets = agent_schema.secrets or agent_schema.tool_exec_environment_variables
        if agent_secrets:
            agent_schema.tool_exec_environment_variables = {key: "" for key in agent_secrets}
            agent_schema.secrets = {key: "" for key in agent_secrets}

        if agent_schema.messages:
            for message in agent_schema.messages:
                message_file_id = self._map_db_to_file_id(message.id, MessageSchema.__id_prefix__)
                message.id = message_file_id
                message.agent_id = agent_file_id

        if agent_schema.in_context_message_ids:
            agent_schema.in_context_message_ids = [
                self._map_db_to_file_id(message_id, MessageSchema.__id_prefix__, allow_new=False)
                for message_id in agent_schema.in_context_message_ids
            ]

        if agent_schema.tool_ids:
            agent_schema.tool_ids = [self._map_db_to_file_id(tool_id, ToolSchema.__id_prefix__) for tool_id in agent_schema.tool_ids]

        if agent_schema.source_ids:
            agent_schema.source_ids = [
                self._map_db_to_file_id(source_id, SourceSchema.__id_prefix__) for source_id in agent_schema.source_ids
            ]

        if agent_schema.block_ids:
            agent_schema.block_ids = [self._map_db_to_file_id(block_id, BlockSchema.__id_prefix__) for block_id in agent_schema.block_ids]

        if agent_schema.files_agents:
            for file_agent in agent_schema.files_agents:
                file_agent.file_id = self._map_db_to_file_id(file_agent.file_id, FileSchema.__id_prefix__)
                file_agent.source_id = self._map_db_to_file_id(file_agent.source_id, SourceSchema.__id_prefix__)
                file_agent.agent_id = agent_file_id

        if agent_schema.group_ids:
            agent_schema.group_ids = [self._map_db_to_file_id(group_id, GroupSchema.__id_prefix__) for group_id in agent_schema.group_ids]

        return agent_schema

    def _convert_tool_to_schema(self, tool) -> ToolSchema:
        """Convert Tool to ToolSchema with ID remapping"""
        tool_file_id = self._map_db_to_file_id(tool.id, ToolSchema.__id_prefix__, allow_new=False)
        tool_schema = ToolSchema.from_tool(tool)
        tool_schema.id = tool_file_id
        return tool_schema

    def _convert_block_to_schema(self, block) -> BlockSchema:
        """Convert Block to BlockSchema with ID remapping"""
        block_file_id = self._map_db_to_file_id(block.id, BlockSchema.__id_prefix__, allow_new=False)
        block_schema = BlockSchema.from_block(block)
        block_schema.id = block_file_id
        return block_schema

    def _convert_source_to_schema(self, source) -> SourceSchema:
        """Convert Source to SourceSchema with ID remapping"""
        source_file_id = self._map_db_to_file_id(source.id, SourceSchema.__id_prefix__, allow_new=False)
        source_schema = SourceSchema.from_source(source)
        source_schema.id = source_file_id
        return source_schema

    def _convert_file_to_schema(self, file_metadata) -> FileSchema:
        """Convert FileMetadata to FileSchema with ID remapping"""
        file_file_id = self._map_db_to_file_id(file_metadata.id, FileSchema.__id_prefix__, allow_new=False)
        file_schema = FileSchema.from_file_metadata(file_metadata)
        file_schema.id = file_file_id
        file_schema.source_id = self._map_db_to_file_id(file_metadata.source_id, SourceSchema.__id_prefix__, allow_new=False)
        return file_schema

    async def _extract_unique_mcp_servers(self, tools: List, actor: User) -> List:
        """Extract unique MCP servers from tools based on metadata, using server_id if available, otherwise falling back to server_name."""
        mcp_server_ids = set()
        mcp_server_names = set()
        for tool in tools:
            # Check if tool has MCP metadata
            if tool.metadata_ and MCP_TOOL_TAG_NAME_PREFIX in tool.metadata_:
                mcp_metadata = tool.metadata_[MCP_TOOL_TAG_NAME_PREFIX]
                # TODO: @jnjpng clean this up once we fully migrate to server_id being the main identifier
                if "server_id" in mcp_metadata:
                    mcp_server_ids.add(mcp_metadata["server_id"])
                elif "server_name" in mcp_metadata:
                    mcp_server_names.add(mcp_metadata["server_name"])

        # Fetch MCP servers by ID
        mcp_servers = []
        fetched_server_ids = set()
        if mcp_server_ids:
            try:
                mcp_servers = await self.mcp_manager.get_mcp_servers_by_ids(list(mcp_server_ids), actor)
                fetched_server_ids.update([mcp_server.id for mcp_server in mcp_servers])
            except Exception as e:
                logger.warning(f"Failed to fetch MCP servers by IDs {mcp_server_ids}: {e}")

        # Fetch MCP servers by name if not already fetched by ID
        if mcp_server_names:
            for server_name in mcp_server_names:
                try:
                    mcp_server = await self.mcp_manager.get_mcp_server(server_name, actor)
                    if mcp_server and mcp_server.id not in fetched_server_ids:
                        mcp_servers.append(mcp_server)
                except Exception as e:
                    logger.warning(f"Failed to fetch MCP server by name {server_name}: {e}")

        return mcp_servers

    def _convert_mcp_server_to_schema(self, mcp_server: MCPServer) -> MCPServerSchema:
        """Convert MCPServer to MCPServerSchema with ID remapping and auth scrubbing"""
        try:
            mcp_file_id = self._map_db_to_file_id(mcp_server.id, MCPServerSchema.__id_prefix__, allow_new=False)
            mcp_schema = MCPServerSchema.from_mcp_server(mcp_server)
            mcp_schema.id = mcp_file_id
            return mcp_schema
        except Exception as e:
            logger.error(f"Failed to convert MCP server {mcp_server.id}: {e}")
            raise

    def _convert_group_to_schema(self, group: Group) -> GroupSchema:
        """Convert Group to GroupSchema with ID remapping"""
        try:
            group_file_id = self._map_db_to_file_id(group.id, GroupSchema.__id_prefix__, allow_new=False)
            group_schema = GroupSchema.from_group(group)
            group_schema.id = group_file_id
            group_schema.agent_ids = [
                self._map_db_to_file_id(agent_id, AgentSchema.__id_prefix__, allow_new=False) for agent_id in group_schema.agent_ids
            ]
            if hasattr(group_schema.manager_config, "manager_agent_id"):
                group_schema.manager_config.manager_agent_id = self._map_db_to_file_id(
                    group_schema.manager_config.manager_agent_id, AgentSchema.__id_prefix__, allow_new=False
                )
            return group_schema
        except Exception as e:
            logger.error(f"Failed to convert group {group.id}: {e}")
            raise

    async def export(self, agent_ids: List[str], actor: User) -> AgentFileSchema:
        """
        Export agents and their related entities to AgentFileSchema format.

        Args:
            agent_ids: List of agent UUIDs to export

        Returns:
            AgentFileSchema with all related entities

        Raises:
            AgentFileExportError: If export fails
        """
        try:
            self._reset_state()

            agent_states = await self.agent_manager.get_agents_by_ids_async(agent_ids=agent_ids, actor=actor)

            # Validate that all requested agents were found
            if len(agent_states) != len(agent_ids):
                found_ids = {agent.id for agent in agent_states}
                missing_ids = [agent_id for agent_id in agent_ids if agent_id not in found_ids]
                raise AgentNotFoundForExportError(missing_ids)

            groups = []
            group_agent_ids = []
            for agent_state in agent_states:
                if agent_state.multi_agent_group != None:
                    groups.append(agent_state.multi_agent_group)
                    group_agent_ids.extend(agent_state.multi_agent_group.agent_ids)

            group_agent_ids = list(set(group_agent_ids) - set(agent_ids))
            if group_agent_ids:
                group_agent_states = await self.agent_manager.get_agents_by_ids_async(agent_ids=group_agent_ids, actor=actor)
                if len(group_agent_states) != len(group_agent_ids):
                    found_ids = {agent.id for agent in group_agent_states}
                    missing_ids = [agent_id for agent_id in group_agent_ids if agent_id not in found_ids]
                    raise AgentFileExportError(f"The following agent IDs were not found: {missing_ids}")
                agent_ids.extend(group_agent_ids)
                agent_states.extend(group_agent_states)

            # cache for file-agent relationships to avoid duplicate queries
            files_agents_cache = {}  # Maps agent_id to list of file_agent relationships

            # Extract unique entities across all agents
            tool_set = self._extract_unique_tools(agent_states)
            block_set = self._extract_unique_blocks(agent_states)

            # Extract MCP servers from tools BEFORE conversion (must be done before ID mapping)
            mcp_server_set = await self._extract_unique_mcp_servers(tool_set, actor)

            # Map MCP server IDs before converting schemas
            for mcp_server in mcp_server_set:
                self._map_db_to_file_id(mcp_server.id, MCPServerSchema.__id_prefix__)

            # Extract sources and files from agent states BEFORE conversion (with caching)
            source_set, file_set = await self._extract_unique_sources_and_files_from_agents(agent_states, actor, files_agents_cache)

            # Convert to schemas with ID remapping (reusing cached file-agent data)
            agent_schemas = [
                await self._convert_agent_state_to_schema(agent_state, actor=actor, files_agents_cache=files_agents_cache)
                for agent_state in agent_states
            ]
            tool_schemas = [self._convert_tool_to_schema(tool) for tool in tool_set]
            block_schemas = [self._convert_block_to_schema(block) for block in block_set]
            source_schemas = [self._convert_source_to_schema(source) for source in source_set]
            file_schemas = [self._convert_file_to_schema(file_metadata) for file_metadata in file_set]
            mcp_server_schemas = [self._convert_mcp_server_to_schema(mcp_server) for mcp_server in mcp_server_set]
            group_schemas = [self._convert_group_to_schema(group) for group in groups]

            logger.info(f"Exporting {len(agent_ids)} agents to agent file format")

            # Return AgentFileSchema with converted entities
            return AgentFileSchema(
                agents=agent_schemas,
                groups=group_schemas,
                blocks=block_schemas,
                files=file_schemas,
                sources=source_schemas,
                tools=tool_schemas,
                mcp_servers=mcp_server_schemas,
                metadata={"revision_id": await get_latest_alembic_revision()},
                created_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Failed to export agent file: {e}")
            raise AgentExportProcessingError(str(e), e) from e

    async def import_file(
        self,
        schema: AgentFileSchema,
        actor: User,
        append_copy_suffix: bool = False,
        override_name: Optional[str] = None,
        override_existing_tools: bool = True,
        dry_run: bool = False,
        env_vars: Optional[Dict[str, Any]] = None,
        override_embedding_config: Optional[EmbeddingConfig] = None,
        project_id: Optional[str] = None,
    ) -> ImportResult:
        """
        Import AgentFileSchema into the database.

        Args:
            schema: The agent file schema to import
            dry_run: If True, validate but don't commit changes

        Returns:
            ImportResult with success status and details

        Raises:
            AgentFileImportError: If import fails
        """
        try:
            self._reset_state()

            if dry_run:
                logger.info("Starting dry run import validation")
            else:
                logger.info("Starting agent file import")

            # Validate schema first
            self._validate_schema(schema)

            if dry_run:
                return ImportResult(
                    success=True,
                    message="Dry run validation passed",
                    imported_count=0,
                )

            # Import in dependency order
            imported_count = 0
            file_to_db_ids = {}  # Maps file IDs to new database IDs
            # in-memory cache for file metadata to avoid repeated db calls
            file_metadata_cache = {}  # Maps database file ID to FileMetadata

            # 1. Create MCP servers first (tools depend on them)
            if schema.mcp_servers:
                for mcp_server_schema in schema.mcp_servers:
                    server_data = mcp_server_schema.model_dump(exclude={"id"})
                    filtered_server_data = self._filter_dict_for_model(server_data, MCPServer)
                    create_schema = MCPServer(**filtered_server_data)

                    # Note: We don't have auth info from export, so the user will need to re-configure auth.
                    # TODO: @jnjpng store metadata about obfuscated metadata to surface to the user
                    created_mcp_server = await self.mcp_manager.create_or_update_mcp_server(create_schema, actor)
                    file_to_db_ids[mcp_server_schema.id] = created_mcp_server.id
                    imported_count += 1

            # 2. Create tools (may depend on MCP servers) - using bulk upsert for efficiency
            if schema.tools:
                # convert tool schemas to pydantic tools
                pydantic_tools = []
                for tool_schema in schema.tools:
                    pydantic_tools.append(Tool(**tool_schema.model_dump(exclude={"id"})))

                # bulk upsert all tools at once
                created_tools = await self.tool_manager.bulk_upsert_tools_async(
                    pydantic_tools, actor, override_existing_tools=override_existing_tools
                )

                # map file ids to database ids
                # note: tools are matched by name during upsert, so we need to match by name here too
                created_tools_by_name = {tool.name: tool for tool in created_tools}
                for tool_schema in schema.tools:
                    created_tool = created_tools_by_name.get(tool_schema.name)
                    if created_tool:
                        file_to_db_ids[tool_schema.id] = created_tool.id
                        imported_count += 1
                    else:
                        logger.warning(f"Tool {tool_schema.name} was not created during bulk upsert")

            # 2. Create blocks (no dependencies) - using batch create for efficiency
            if schema.blocks:
                # convert block schemas to pydantic blocks (excluding IDs to create new blocks)
                pydantic_blocks = []
                for block_schema in schema.blocks:
                    pydantic_blocks.append(Block(**block_schema.model_dump(exclude={"id"})))

                # batch create all blocks at once
                created_blocks = await self.block_manager.batch_create_blocks_async(pydantic_blocks, actor)

                # map file ids to database ids
                for block_schema, created_block in zip(schema.blocks, created_blocks):
                    file_to_db_ids[block_schema.id] = created_block.id
                    imported_count += 1

            # 3. Create sources (no dependencies) - using bulk upsert for efficiency
            if schema.sources:
                # convert source schemas to pydantic sources
                pydantic_sources = []

                # First, do a fast batch check for existing source names to avoid conflicts
                source_names_to_check = [s.name for s in schema.sources]
                existing_source_names = await self.source_manager.get_existing_source_names(source_names_to_check, actor)

                # override embedding_config
                if override_embedding_config:
                    for source_schema in schema.sources:
                        source_schema.embedding_config = override_embedding_config
                        source_schema.embedding = override_embedding_config.handle

                for source_schema in schema.sources:
                    source_data = source_schema.model_dump(exclude={"id", "embedding", "embedding_chunk_size"})

                    # Check if source name already exists, if so add unique suffix
                    original_name = source_data["name"]
                    if original_name in existing_source_names:
                        unique_suffix = uuid.uuid4().hex[:8]
                        source_data["name"] = f"{original_name}_{unique_suffix}"

                    pydantic_sources.append(Source(**source_data))

                # bulk upsert all sources at once
                created_sources = await self.source_manager.bulk_upsert_sources_async(pydantic_sources, actor)

                # map file ids to database ids
                # note: sources are matched by name during upsert, so we need to match by name here too
                created_sources_by_name = {source.name: source for source in created_sources}
                for i, source_schema in enumerate(schema.sources):
                    # Use the pydantic source name (which may have been modified for uniqueness)
                    source_name = pydantic_sources[i].name
                    created_source = created_sources_by_name.get(source_name)
                    if created_source:
                        file_to_db_ids[source_schema.id] = created_source.id
                        imported_count += 1
                    else:
                        logger.warning(f"Source {source_name} was not created during bulk upsert")

            # 4. Create files (depends on sources)
            for file_schema in schema.files:
                # Convert FileSchema back to FileMetadata
                file_data = file_schema.model_dump(exclude={"id", "content"})
                # Remap source_id from file ID to database ID
                file_data["source_id"] = file_to_db_ids[file_schema.source_id]
                # Set processing status to PARSING since we have parsed content but need to re-embed
                file_data["processing_status"] = FileProcessingStatus.PARSING
                file_data["error_message"] = None
                file_data["total_chunks"] = None
                file_data["chunks_embedded"] = None
                file_metadata = FileMetadata(**file_data)
                created_file = await self.file_manager.create_file(file_metadata, actor, text=file_schema.content)
                file_to_db_ids[file_schema.id] = created_file.id
                imported_count += 1

            # 5. Process files for chunking/embedding (depends on files and sources)
            # Start background tasks for file processing
            background_tasks = []
            if schema.files and any(f.content for f in schema.files):
                # Use override embedding config if provided, otherwise use agent's config
                embedder_config = override_embedding_config if override_embedding_config else schema.agents[0].embedding_config
                # determine which embedder to use - turbopuffer takes precedence
                if should_use_tpuf():
                    from letta.services.file_processor.embedder.turbopuffer_embedder import TurbopufferEmbedder

                    embedder = TurbopufferEmbedder(embedding_config=embedder_config)
                elif should_use_pinecone():
                    embedder = PineconeEmbedder(embedding_config=embedder_config)
                else:
                    embedder = OpenAIEmbedder(embedding_config=embedder_config)
                file_processor = FileProcessor(
                    file_parser=self.file_parser,
                    embedder=embedder,
                    actor=actor,
                )

                for file_schema in schema.files:
                    if file_schema.content:  # Only process files with content
                        file_db_id = file_to_db_ids[file_schema.id]
                        source_db_id = file_to_db_ids[file_schema.source_id]

                        # Get the created file metadata (with caching)
                        if file_db_id not in file_metadata_cache:
                            file_metadata_cache[file_db_id] = await self.file_manager.get_file_by_id(file_db_id, actor)
                        file_metadata = file_metadata_cache[file_db_id]

                        # Save the db call of fetching content again
                        file_metadata.content = file_schema.content

                        # Create background task for file processing
                        # TODO: This can be moved to celery or RQ or something
                        task = safe_create_task(
                            self._process_file_async(
                                file_metadata=file_metadata, source_id=source_db_id, file_processor=file_processor, actor=actor
                            ),
                            label=f"process_file_{file_metadata.file_name}",
                        )
                        background_tasks.append(task)
                        logger.info(f"Started background processing for file {file_metadata.file_name} (ID: {file_db_id})")

            # 6. Create agents with empty message history
            for agent_schema in schema.agents:
                # Override embedding_config if provided
                if override_embedding_config:
                    agent_schema.embedding_config = override_embedding_config
                    agent_schema.embedding = override_embedding_config.handle

                # Convert AgentSchema back to CreateAgent, remapping tool/block IDs
                agent_data = agent_schema.model_dump(exclude={"id", "in_context_message_ids", "messages"})

                # Handle agent name override: override_name takes precedence over append_copy_suffix
                if override_name:
                    agent_data["name"] = override_name
                elif append_copy_suffix:
                    agent_data["name"] = agent_data.get("name") + "_copy"

                # Remap tool_ids from file IDs to database IDs
                if agent_data.get("tool_ids"):
                    agent_data["tool_ids"] = [file_to_db_ids[file_id] for file_id in agent_data["tool_ids"]]

                # Remap block_ids from file IDs to database IDs
                if agent_data.get("block_ids"):
                    agent_data["block_ids"] = [file_to_db_ids[file_id] for file_id in agent_data["block_ids"]]

                # Remap source_ids from file IDs to database IDs
                if agent_data.get("source_ids"):
                    agent_data["source_ids"] = [file_to_db_ids[file_id] for file_id in agent_data["source_ids"]]

                if env_vars and agent_data.get("secrets"):
                    # update environment variable values from the provided env_vars dict
                    for key in agent_data["secrets"]:
                        agent_data["secrets"][key] = env_vars.get(key, "")
                        agent_data["tool_exec_environment_variables"][key] = env_vars.get(key, "")
                elif env_vars and agent_data.get("tool_exec_environment_variables"):
                    # also handle tool_exec_environment_variables for backwards compatibility
                    for key in agent_data["tool_exec_environment_variables"]:
                        agent_data["tool_exec_environment_variables"][key] = env_vars.get(key, "")
                        agent_data["secrets"][key] = env_vars.get(key, "")

                # Override project_id if provided
                if project_id:
                    agent_data["project_id"] = project_id

                agent_create = CreateAgent(**agent_data)
                created_agent = await self.agent_manager.create_agent_async(agent_create, actor, _init_with_no_messages=True)
                file_to_db_ids[agent_schema.id] = created_agent.id
                imported_count += 1

            # 7. Create messages and update agent message_ids
            for agent_schema in schema.agents:
                agent_db_id = file_to_db_ids[agent_schema.id]
                message_file_to_db_ids = {}

                # Create messages for this agent
                messages = []
                for message_schema in agent_schema.messages:
                    # Convert MessageSchema back to Message, setting agent_id to new DB ID
                    message_data = message_schema.model_dump(exclude={"id", "type"})
                    message_data["agent_id"] = agent_db_id  # Remap agent_id to new database ID
                    message_obj = Message(**message_data)
                    messages.append(message_obj)
                    # Map file ID to the generated database ID immediately
                    message_file_to_db_ids[message_schema.id] = message_obj.id

                created_messages = await self.message_manager.create_many_messages_async(
                    pydantic_msgs=messages,
                    actor=actor,
                    project_id=created_agent.project_id,
                    template_id=created_agent.template_id,
                )
                imported_count += len(created_messages)

                # Remap in_context_message_ids from file IDs to database IDs
                in_context_db_ids = [message_file_to_db_ids[message_schema_id] for message_schema_id in agent_schema.in_context_message_ids]

                # Update agent with the correct message_ids
                await self.agent_manager.update_message_ids_async(agent_id=agent_db_id, message_ids=in_context_db_ids, actor=actor)

            # 8. Create file-agent relationships (depends on agents and files)
            for agent_schema in schema.agents:
                if agent_schema.files_agents:
                    agent_db_id = file_to_db_ids[agent_schema.id]

                    # Prepare files for bulk attachment
                    files_for_agent = []
                    visible_content_map = {}

                    for file_agent_schema in agent_schema.files_agents:
                        file_db_id = file_to_db_ids[file_agent_schema.file_id]

                        # Use cached file metadata if available (with content)
                        if file_db_id not in file_metadata_cache:
                            file_metadata_cache[file_db_id] = await self.file_manager.get_file_by_id(
                                file_db_id, actor, include_content=True
                            )
                        file_metadata = file_metadata_cache[file_db_id]
                        files_for_agent.append(file_metadata)

                        if file_agent_schema.visible_content:
                            visible_content_map[file_metadata.file_name] = file_agent_schema.visible_content

                    # Bulk attach files to agent
                    await self.file_agent_manager.attach_files_bulk(
                        agent_id=agent_db_id,
                        files_metadata=files_for_agent,
                        visible_content_map=visible_content_map,
                        actor=actor,
                        max_files_open=agent_schema.max_files_open,
                    )
                    imported_count += len(files_for_agent)

            # Extract the imported agent IDs (database IDs)
            imported_agent_ids = []
            for agent_schema in schema.agents:
                if agent_schema.id in file_to_db_ids:
                    imported_agent_ids.append(file_to_db_ids[agent_schema.id])

            for group in schema.groups:
                group_data = group.model_dump(exclude={"id"})
                group_data["agent_ids"] = [file_to_db_ids[agent_id] for agent_id in group_data["agent_ids"]]
                if "manager_agent_id" in group_data["manager_config"]:
                    group_data["manager_config"]["manager_agent_id"] = file_to_db_ids[group_data["manager_config"]["manager_agent_id"]]
                created_group = await self.group_manager.create_group_async(GroupCreate(**group_data), actor)
                file_to_db_ids[group.id] = created_group.id
                imported_count += 1

            # prepare result message
            num_background_tasks = len(background_tasks)
            if num_background_tasks > 0:
                message = (
                    f"Import completed successfully. Imported {imported_count} entities. "
                    f"{num_background_tasks} file(s) are being processed in the background for embeddings."
                )
            else:
                message = f"Import completed successfully. Imported {imported_count} entities."

            return ImportResult(
                success=True,
                message=message,
                imported_count=imported_count,
                imported_agent_ids=imported_agent_ids,
                id_mappings=file_to_db_ids,
            )

        except Exception as e:
            logger.exception(f"Failed to import agent file: {e}")
            raise AgentFileImportError(f"Import failed: {e}") from e

    def _validate_id_format(self, schema: AgentFileSchema) -> List[str]:
        """Validate that all IDs follow the expected format"""
        errors = []

        # Define entity types and their expected prefixes
        entity_checks = [
            (schema.agents, AgentSchema.__id_prefix__),
            (schema.groups, GroupSchema.__id_prefix__),
            (schema.blocks, BlockSchema.__id_prefix__),
            (schema.files, FileSchema.__id_prefix__),
            (schema.sources, SourceSchema.__id_prefix__),
            (schema.tools, ToolSchema.__id_prefix__),
            (schema.mcp_servers, MCPServerSchema.__id_prefix__),
        ]

        for entities, expected_prefix in entity_checks:
            for entity in entities:
                if not entity.id.startswith(f"{expected_prefix}-"):
                    errors.append(f"Invalid ID format: {entity.id} should start with '{expected_prefix}-'")
                else:
                    # Check that the suffix is a valid integer
                    try:
                        suffix = entity.id[len(expected_prefix) + 1 :]
                        int(suffix)
                    except ValueError:
                        errors.append(f"Invalid ID format: {entity.id} should have integer suffix")

        # Also check message IDs within agents
        for agent in schema.agents:
            for message in agent.messages:
                if not message.id.startswith(f"{MessageSchema.__id_prefix__}-"):
                    errors.append(f"Invalid message ID format: {message.id} should start with '{MessageSchema.__id_prefix__}-'")
                else:
                    # Check that the suffix is a valid integer
                    try:
                        suffix = message.id[len(MessageSchema.__id_prefix__) + 1 :]
                        int(suffix)
                    except ValueError:
                        errors.append(f"Invalid message ID format: {message.id} should have integer suffix")

        return errors

    def _validate_duplicate_ids(self, schema: AgentFileSchema) -> List[str]:
        """Validate that there are no duplicate IDs within or across entity types"""
        errors = []
        all_ids = set()

        # Check each entity type for internal duplicates and collect all IDs
        entity_collections = [
            ("agents", schema.agents),
            ("groups", schema.groups),
            ("blocks", schema.blocks),
            ("files", schema.files),
            ("sources", schema.sources),
            ("tools", schema.tools),
            ("mcp_servers", schema.mcp_servers),
        ]

        for entity_type, entities in entity_collections:
            entity_ids = [entity.id for entity in entities]

            # Check for duplicates within this entity type
            seen = set()
            duplicates = set()
            for entity_id in entity_ids:
                if entity_id in seen:
                    duplicates.add(entity_id)
                else:
                    seen.add(entity_id)

            if duplicates:
                errors.append(f"Duplicate {entity_type} IDs found: {duplicates}")

            # Check for duplicates across all entity types
            for entity_id in entity_ids:
                if entity_id in all_ids:
                    errors.append(f"Duplicate ID across entity types: {entity_id}")
                all_ids.add(entity_id)

        # Also check message IDs within agents
        for agent in schema.agents:
            message_ids = [msg.id for msg in agent.messages]

            # Check for duplicates within agent messages
            seen = set()
            duplicates = set()
            for message_id in message_ids:
                if message_id in seen:
                    duplicates.add(message_id)
                else:
                    seen.add(message_id)

            if duplicates:
                errors.append(f"Duplicate message IDs in agent {agent.id}: {duplicates}")

            # Check for duplicates across all entity types
            for message_id in message_ids:
                if message_id in all_ids:
                    errors.append(f"Duplicate ID across entity types: {message_id}")
                all_ids.add(message_id)

        return errors

    def _validate_file_source_references(self, schema: AgentFileSchema) -> List[str]:
        """Validate that all file source_id references exist"""
        errors = []
        source_ids = {source.id for source in schema.sources}

        for file in schema.files:
            if file.source_id not in source_ids:
                errors.append(f"File {file.id} references non-existent source {file.source_id}")

        return errors

    def _validate_file_agent_references(self, schema: AgentFileSchema) -> List[str]:
        """Validate that all file-agent relationships reference existing entities"""
        errors = []
        file_ids = {file.id for file in schema.files}
        source_ids = {source.id for source in schema.sources}
        {agent.id for agent in schema.agents}

        for agent in schema.agents:
            for file_agent in agent.files_agents:
                if file_agent.file_id not in file_ids:
                    errors.append(f"File-agent relationship references non-existent file {file_agent.file_id}")
                if file_agent.source_id not in source_ids:
                    errors.append(f"File-agent relationship references non-existent source {file_agent.source_id}")
                if file_agent.agent_id != agent.id:
                    errors.append(f"File-agent relationship has mismatched agent_id {file_agent.agent_id} vs {agent.id}")

        return errors

    def _validate_schema(self, schema: AgentFileSchema):
        """
        Validate the agent file schema for consistency and referential integrity.

        Args:
            schema: The schema to validate

        Raises:
            AgentFileImportError: If validation fails
        """
        errors = []

        # 1. ID Format Validation
        errors.extend(self._validate_id_format(schema))

        # 2. Duplicate ID Detection
        errors.extend(self._validate_duplicate_ids(schema))

        # 3. File Source Reference Validation
        errors.extend(self._validate_file_source_references(schema))

        # 4. File-Agent Reference Validation
        errors.extend(self._validate_file_agent_references(schema))

        if errors:
            raise AgentFileImportError(f"Schema validation failed: {'; '.join(errors)}")

        logger.info("Schema validation passed")

    def _filter_dict_for_model(self, data: dict, model_cls):
        """Filter a dictionary to only include keys that are in the model fields"""
        try:
            allowed = model_cls.model_fields.keys()  # Pydantic v2
        except AttributeError:
            allowed = model_cls.__fields__.keys()  # Pydantic v1
        return {k: v for k, v in data.items() if k in allowed}

    async def _process_file_async(self, file_metadata: FileMetadata, source_id: str, file_processor: FileProcessor, actor: User):
        """
        Process a file asynchronously in the background.

        This method handles chunking and embedding of file content without blocking
        the main import process.

        Args:
            file_metadata: The file metadata with content
            source_id: The database ID of the source
            file_processor: The file processor instance to use
            actor: The user performing the action
        """
        file_id = file_metadata.id
        file_name = file_metadata.file_name

        try:
            logger.info(f"Starting background processing for file {file_name} (ID: {file_id})")

            # process the file for chunking/embedding
            passages = await file_processor.process_imported_file(file_metadata=file_metadata, source_id=source_id)

            logger.info(f"Successfully processed file {file_name} with {len(passages)} passages")

            # file status is automatically updated to COMPLETED by process_imported_file
            return passages

        except Exception as e:
            logger.error(f"Failed to process file {file_name} (ID: {file_id}) in background: {e}")

            # update file status to ERROR
            try:
                await self.file_manager.update_file_status(
                    file_id=file_id,
                    actor=actor,
                    processing_status=FileProcessingStatus.ERROR,
                    error_message=str(e) if str(e) else f"Agent serialization failed: {type(e).__name__}",
                )
            except Exception as update_error:
                logger.error(f"Failed to update file status to ERROR for {file_id}: {update_error}")

            # we don't re-raise here since this is a background task
            # the file will be marked as ERROR and the import can continue

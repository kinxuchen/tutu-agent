from typing import override, Optional, Sequence, Tuple, Any

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, ChannelVersions, \
    CheckpointTuple
import jsonpickle
from langchain_core.runnables import RunnableConfig
from redis import Redis
from redis.asyncio import Redis as RedisAsync


def encode_data(
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
):
    data = {
        "config": config,
        "checkpoint": checkpoint,
        "metadata": metadata,
        "new_versions": new_versions,
    }
    try:
        return jsonpickle.encode(data)
    except Exception as e:
        return None


class RedisCheckpointSaver(BaseCheckpointSaver):
    def __init__(self, redis_client: Redis, prefix):
        self.redis_client = redis_client
        self.key = prefix

    def get_key(self, thread_id):
        return f"{self.key}:{thread_id}"

    @override
    def put(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
    ) -> RunnableConfig | None:
        thread_id = config['configurable']['thread_id']
        user_id = config['configurable']['user_id']
        if thread_id is None:
            pass
        else:
            key = self.get_key("{}:{}".format(user_id,thread_id))
            data_json = encode_data(
                config,
                checkpoint,
                metadata,
                new_versions
            )
            self.redis_client.set(key, data_json)
            return config
    @override
    async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
    ) -> RunnableConfig | None:
        thread_id = config['configurable']['thread_id']
        user_id = config['configurable']['user_id']
        if thread_id is None:
            pass
        else:
            key = self.get_key("{}:{}".format(user_id,thread_id))
            data_json = encode_data(
                config,
                checkpoint,
                metadata,
                new_versions
            )
            self.redis_client.set(key, data_json)
            return config

    @override
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config['configurable']['thread_id']
        user_id = config['configurable']['user_id']
        if thread_id is None:
            return CheckpointTuple(
                config=config
            )
        else:
            key = self.get_key("{}:{}".format(user_id,thread_id))
            data_json = self.redis_client.get(key)
            if data_json is None:
                return None
            else:
                data = jsonpickle.decode(data_json)
                return CheckpointTuple(
                    checkpoint=data.get("checkpoint"),
                    config=data.get("config"),
                    metadata=data.get("metadata")
                )

    @override
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config['configurable']['thread_id']
        user_id = config['configurable']['user_id']
        if thread_id is None:
            return CheckpointTuple(
                config=config
            )
        else:
            key = self.get_key("{}:{}".format(user_id,thread_id))
            data_json = self.redis_client.get(key)
            if data_json is None:
                return None
            else:
                data = jsonpickle.decode(data_json)
                return CheckpointTuple(
                    checkpoint=data.get("checkpoint"),
                    config=data.get("config"),
                    metadata=data.get("metadata")
                )
    @override
    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        # 暂时不处理保存每个超级步骤的临时状态
        return None

    @override
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        # 暂时不处理保存每个超级步骤的临时状态
        return None

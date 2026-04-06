import json

from bellm.dataset.dataset_sources import DatasetSourceHuggingFace


def oasst_adapter(data):
    items = [x for x in data]
    items_map = {x["message_id"]: {"children": [], "text": x['text'], "lang": x["lang"], "role": x["role"]} for x in items}

    heads = []

    for item in items:
        if item["parent_id"] is None:
            heads.append(items_map[item["message_id"]])
        else:
            item_chain = items_map[item["message_id"]]
            items_map[item["parent_id"]]["children"].append(item_chain)

    # Aim for english convos only
    heads = [x for x in heads if x["lang"] == "en"]

    # Traverse the tree forming the conversations
    conversations = []
    def traverse_head(items, conversation_chain):
        if len(items) == 0:
            conversations.append(conversation_chain)

        for item in items:
            traverse_head(
                item['children'],
                [*conversation_chain, {
                    "message": item["text"],
                    "role": {
                        "prompter": "user",
                        "assistant": "assistant"
                    }[item["role"]]
                }]
            )

    # Trigger breath first search
    traverse_head(heads, [])

    conversations = [json.dumps(x) for x in conversations]

    return conversations


OPEN_ASSISTANT_OASST2 = DatasetSourceHuggingFace(
    path="OpenAssistant/oasst2",
    split="train",
    adapter=oasst_adapter
)


INSTRUCTION_DATASETS = [
    OPEN_ASSISTANT_OASST2,
]

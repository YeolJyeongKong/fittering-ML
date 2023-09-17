import random
import pymilvus
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


def connect(host, port):
    connections.connect("default", host=host, port=port)


def disconnect():
    connections.disconnect("default")


def exist_collection(collection_name):
    return utility.has_collection(collection_name)


def delete_collection(collection_name):
    if exist_collection(collection_name):
        utility.drop_collection(collection_name)


def add_vector(collection_name, embedded, product_id, gender):
    image_embedding_collection = Collection(collection_name)
    image_embedding_collection.load()
    entities = [
        product_id,
        gender,
        embedded,
    ]
    try:
        image_embedding_collection.insert(entities)
        image_embedding_collection.flush()

    except pymilvus.exceptions.ParamError:
        delete_collection(collection_name)
        save_collection(collection_name, embedded, product_id, gender)


def save_collection(collection_name, embedded, product_id, gender):
    fields = [
        FieldSchema(
            name="product_id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(name="gender", dtype=DataType.VARCHAR, max_length=1),
        FieldSchema(
            name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=embedded.shape[1]
        ),
    ]
    schema = CollectionSchema(fields, "product_image_vector")

    image_embedding_collection = Collection(
        collection_name, schema, consistency_level="Strong"
    )

    entities = [
        product_id,
        gender,
        embedded,
    ]
    image_embedding_collection.insert(entities)

    image_embedding_collection.flush()

    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }

    image_embedding_collection.create_index("embeddings", index)


def search_vector(collection_name, product_ids, gender, top_k, recommendation_n):
    image_embedding_collection = Collection(collection_name)
    image_embedding_collection.load()

    res = image_embedding_collection.query(
        expr=f"product_id in {product_ids}",
        output_fields=["product_id", "gender", "embeddings"],
    )
    search_embeddings = [r["embeddings"] for r in res]

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    result = image_embedding_collection.search(
        search_embeddings,
        "embeddings",
        search_params,
        limit=top_k,
        expr=f"product_id not in {product_ids} and gender in ['{gender}', 'A']",
        output_fields=["product_id"],
    )

    search_result = []
    for hits in result:
        for hit in hits:
            search_result += [hit.entity.get("product_id")]
    random.shuffle(search_result)

    return search_result[:recommendation_n]

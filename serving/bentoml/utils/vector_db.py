import random
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

connections.connect("default", host="localhost", port="19530")


def save_vector(embedded, product_id, gender):
    connections.connect("default", host="localhost", port="19530")
    if utility.has_collection("image_vector_db"):
        utility.drop_collection("image_vector_db")

    fields = [
        FieldSchema(
            name="product_id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(name="gender", dtype=DataType.VARCHAR, max_length=1),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=64),
    ]
    schema = CollectionSchema(fields, "product_image_vector")

    image_vector_db = Collection("image_vector_db", schema, consistency_level="Strong")

    entities = [
        product_id,
        gender,
        embedded,
    ]
    insert_result = image_vector_db.insert(entities)

    image_vector_db.flush()

    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }

    image_vector_db.create_index("embeddings", index)
    connections.disconnect("default")


def search_vector(product_ids, gender, top_k, recommendation_n):
    connections.connect("default", host="localhost", port="19530")
    image_vector_db = Collection("image_vector_db")
    image_vector_db.load()

    res = image_vector_db.query(
        expr=f"product_id in {product_ids}",
        output_fields=["product_id", "gender", "embeddings"],
    )
    search_embeddings = [r["embeddings"] for r in res]

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    result = image_vector_db.search(
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

    connections.disconnect("default")
    return search_result[:recommendation_n]

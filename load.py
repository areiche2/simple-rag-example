

import pinecone
import openai

import configs
import wiki


def create_index(pc: pinecone.Pinecone, index_name: str, dimension: int, metric: str):
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=pinecone.ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )
    return pc.Index(name=configs.PINECONE_INDEX)


def upload(pci: pinecone.Index, oai: openai.OpenAI, title: str, text: str):
    print(f"Uploading page: {title}")
    resp = pci.upsert(
        vectors=[
            (title, get_embedding(oai, text), {"text": text})
        ]
    )
    print(f"\t{resp}")


def get_embedding(oai: openai.OpenAI, text: str):
    text = text.replace("\n", " ")
    return (
        oai.embeddings.create(
            input=[text],
            model=configs.OPENAI_EMBEDDING_MODEL
        ).data[0].embedding
    )


def main():
    pc = pinecone.Pinecone()
    oai = openai.OpenAI()
    pci = create_index(
        pc=pc,
        index_name=configs.PINECONE_INDEX,
        dimension=configs.OPENAI_EMBEDDING_DIMENSION,
        metric="cosine"
    )
    # https://en.wikipedia.org/wiki/Renard_R.31
    # https://en.wikipedia.org/wiki/Australia_women%27s_national_softball_team
    for title in ["Renard_R.31", "Australia_women's_national_softball_team"]:
        upload(pci, oai, title, wiki.fetch(title))


if __name__ == '__main__':
    main()

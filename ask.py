import argparse
import textwrap
from typing import List

import openai
import pinecone

import configs


DEBUG = False


def _dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def get_embedding(oai: openai.OpenAI, text: str):
    text = text.replace("\n", " ")
    return (
        oai.embeddings.create(
            input=[text],
            model=configs.OPENAI_EMBEDDING_MODEL
        ).data[0].embedding
    )


def get_response(oai: openai.OpenAI, prompt: str):
    response = oai.chat.completions.create(
        model=configs.OPEN_AI_LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip("\n")


def build_prompt(question: str, source: str):
    return f"""
Your are a helpful assistant. Your goal is to provide a helpful answer to the user's question.
You must answer the question using only the data in the context below. If you cannot answer using
the context, you should respond with "I don't know". The question and context are delimited by the
marker <<<<>>>>
<<<<>>>>
CONTEXT:
{source}

<<<<>>>>
QUESTION
{question}

"""


def query_llm(oai: openai.OpenAI, question: str, source: str):
    _dprint(f"Querying LLM:")
    propmt = build_prompt(question, source)
    _dprint(textwrap.indent(propmt, prefix="  "))
    return get_response(oai, propmt)


def query(question: str, pci: pinecone.Index, oai: openai.OpenAI, debug: bool):
    emb = get_embedding(oai, question)
    _dprint(f"Embedding: {emb[:5]}...")
    docs = pci.query(
        vector=emb,
        top_k=1,
        include_values=True,
        include_metadata=True,
        filter=None
    )
    if not (docs.matches):
        return "Unable to find an answer."
    m = docs.matches[0]
    _dprint(f"Found: {m.id}...")
    text = m.metadata["text"]
    return query_llm(oai, question=question, source=text)


def main(questions: List[str]):
    pci = pinecone.Pinecone().Index(name=configs.PINECONE_INDEX)
    oai = openai.OpenAI()
    for q in questions:
        print("-"*60)
        print(f"Question: {q}")
        answer = query(q, pci, oai, args.debug)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", type=str, nargs="+")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        DEBUG = True
    main(args.questions)

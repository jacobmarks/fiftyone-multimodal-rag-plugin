"""Multimodal RAG Plugin

"""

from bson import json_util
from hashlib import sha256
import os
from pathlib import Path  ## use to add pdfs
import random
import sys
import textwrap

import json
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont

# import qdrant_client

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import (
    Document,
    ImageDocument,
    ImageNode,
    TextNode,
)
from llama_index.core.storage import StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.multi_modal.base import (
    MultiModalVectorStoreIndex,
)
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.multi_modal_llms.gemini.base import GEMINI_MM_MODELS
from llama_index.multi_modal_llms.replicate import ReplicateMultiModal
from llama_index.multi_modal_llms.replicate.base import (
    REPLICATE_MULTI_MODAL_LLM_MODELS,
)
from llama_index.vector_stores.milvus import MilvusVectorStore

import fiftyone as fo
import fiftyone.core.utils as fou
from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.types as fot
import fiftyone.plugins as fop

DEFAULT_MAX_NEW_TOKENS = 300

OPENAI_MODEL_NAMES = ["gpt-4-vision-preview"]
REPLICATE_MODEL_NAMES = list(REPLICATE_MULTI_MODAL_LLM_MODELS.keys())
GEMINI_MODEL_NAMES = GEMINI_MM_MODELS

ALL_MODEL_NAMES = []
ALL_MODEL_NAMES.extend(GEMINI_MODEL_NAMES)
ALL_MODEL_NAMES.extend(OPENAI_MODEL_NAMES)
ALL_MODEL_NAMES.extend(REPLICATE_MODEL_NAMES)
ALL_MODEL_NAMES = sorted(ALL_MODEL_NAMES)

URI = os.environ.get("ZILLIZ_URI", "")
TOKEN = os.environ.get("ZILLIZ_TOKEN", "")

# define our QA prompt template
qa_tmpl_str = (
    "Some images and text are provided as context.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "If the images and text provided cannot help in answering the query\n"
    "then respond that you are unable to answer the query. Otherwise,\n"
    "using only the context provided, and not prior knowledge,\n"
    "provide an answer to the query."
    "Query: {query_str}\n"
    "Answer: "
)
qa_tmpl = PromptTemplate(qa_tmpl_str)

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    # pylint: disable=no-name-in-module,import-error
    from cache_manager import get_cache


def _hash_text(text):
    return sha256(text.encode()).hexdigest()[0:8]


def _create_hash():
    """Create a random hash"""
    randint = random.randint(0, 100000000)
    hash = sha256(str(randint).encode("utf-8")).hexdigest()[:10]
    return hash


def serialize_view(view):
    return json.loads(json_util.dumps(view._serialize()))


def _llama_doc_from_fo_sample(sample, text_field=None):
    llama_doc_dict = {}

    sample_dict = sample.to_dict()
    with open("/tmp/out.txt", "w") as f:
        f.write(str(sample_dict))

    for k, v in sample_dict.items():
        if k.startswith("llama__"):
            llama_doc_dict[k[7:]] = v

    if "metadata" not in llama_doc_dict:
        llama_doc_dict["metadata"] = {}

    llama_doc_dict["metadata"]["fiftyone_sample_id"] = str(sample.id)
    if text_field:
        llama_doc_dict["text"] = sample[text_field]

    if sample_dict["llama__class_name"] == "ImageDocument":
        return ImageDocument.from_dict(llama_doc_dict)
    else:
        return Document.from_dict(llama_doc_dict)


def _is_image_document(document):
    return isinstance(document, ImageDocument)


def _is_pdf_document(document):
    return (
        isinstance(document, Document)
        and document.metadata["file_name"].split(".")[-1] == "pdf"
    )


def _save_pdf_page_to_image(pdf_path, page_number, output_dir="/tmp"):
    basename = os.path.basename(pdf_path).split(".")[0]
    out_path = os.path.join(output_dir, f"{basename}_{page_number}.png")
    if os.path.exists(out_path):
        return out_path
    image = convert_from_path(
        pdf_path, first_page=page_number, last_page=page_number, dpi=100
    )[0]
    image.save(out_path, "PNG")
    return out_path


def _textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height


def _save_text_to_image(text, output_dir="/tmp"):
    out_path = os.path.join(output_dir, f"{_hash_text(text)}.png")
    if os.path.exists(out_path):
        return out_path

    trunc = 1650
    max_image_width = 480
    padding = (20, 20)
    font = ImageFont.load_default()

    # Truncate text if necessary
    truncated_text = (text[:trunc] + "...") if len(text) > trunc else text

    # Calculate approximate character count that fits in the image width
    avg_char_width = _textsize("x" * 100, font=font)[0] / 100
    chars_per_line = max(
        1, int((max_image_width - 2 * padding[0]) / avg_char_width)
    )

    # Wrap the text
    wrapped_text = textwrap.wrap(truncated_text, width=chars_per_line)

    _line_widths = [_textsize(line, font=font)[0] for line in wrapped_text]
    longest_line_width = max(_line_widths) if _line_widths else 100
    image_width = min(
        max_image_width, longest_line_width + 2 * padding[0]
    )  # Adjust width to content
    line_height = (
        _textsize("Ay", font=font)[1] + 10
    )  # Estimate line height with padding
    image_height = padding[1] * 2 + len(wrapped_text) * line_height

    # Create the final image
    img = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(img)

    current_height = padding[1]
    for line in wrapped_text:
        draw.text((padding[0], current_height), line, fill="black", font=font)
        line_height = _textsize(line, font=font)[1]
        current_height += line_height + 10  # Adjust line spacing as needed

    img.save(out_path, "PNG")
    return out_path


def _initialize_llama_index_sample_dict(document):
    sample_dict = {}
    for key, value in document.to_dict().items():
        sample_dict[f"llama__{key}"] = value
    return sample_dict


def _add_image_document_to_dataset(dataset, document):
    sample_dict = _initialize_llama_index_sample_dict(document)
    sample_dict["tags"] = ["image"]
    fp = document.metadata["file_path"]
    sample_dict["filepath"] = os.path.expanduser(fp)
    dataset.add_sample(fo.Sample(**sample_dict))


def _add_pdf_document_to_dataset(dataset, document, output_dir="/tmp"):
    if "file_path" not in document.metadata:
        return _add_text_document_to_dataset(
            dataset, document, output_dir=output_dir
        )

    sample_dict = _initialize_llama_index_sample_dict(document)
    sample_dict["tags"] = ["text"]
    page_num = int(document.metadata["page_label"])
    image_path = _save_pdf_page_to_image(
        document.metadata["file_path"], page_num, output_dir=output_dir
    )
    sample_dict["filepath"] = image_path
    dataset.add_sample(fo.Sample(**sample_dict))


def _add_text_document_to_dataset(dataset, document, output_dir="/tmp"):
    sample_dict = _initialize_llama_index_sample_dict(document)
    sample_dict["tags"] = ["text"]
    image_path = _save_text_to_image(document.text, output_dir=output_dir)
    sample_dict["filepath"] = image_path
    dataset.add_sample(fo.Sample(**sample_dict))


def add_documents_to_dataset(documents, dataset, output_dir="/tmp"):
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for document in documents:
        if _is_image_document(document):
            _add_image_document_to_dataset(dataset, document)
            if "image" not in dataset.tags:
                dataset.tags.append("image")
        else:
            if _is_pdf_document(document):
                _add_pdf_document_to_dataset(
                    dataset, document, output_dir=output_dir
                )
            else:
                _add_text_document_to_dataset(
                    dataset, document, output_dir=output_dir
                )

            if "text" not in dataset.tags:
                dataset.tags.append("text")

    dataset.compute_metadata()


def create_dataset_from_documents(documents, **kwargs):
    output_dir = kwargs.pop("output_dir", "/tmp")

    dataset = fo.Dataset(**kwargs)
    add_documents_to_dataset(documents, dataset, output_dir=output_dir)

    sidebar_groups = fo.DatasetAppConfig.default_sidebar_groups(dataset)
    sidebar_groups[3].paths = [
        p for p in sidebar_groups[3].paths if "llama_" not in p
    ]
    print(sidebar_groups[3].paths)
    dataset.app_config.sidebar_groups = sidebar_groups
    dataset.save()
    return dataset


class CreateDatasetFromLlamaDocuments(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="create_dataset_from_llama_documents",
            label="Create Dataset from LlamaIndex Documents",
            description=(
                "Create a dataset from LlamaIndex `Document` and `ImageDocument` instances"
            ),
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Create Dataset from LlamaIndex Documents",
            description="Create a dataset from LlamaIndex `Document` and `ImageDocument` instances",
        )
        inputs.str(
            "dataset_name",
            label="Dataset name",
            description="The name of the dataset to create",
            required=True,
        )
        inputs.bool(
            "persistent",
            label="Persistent",
            description="Whether the dataset is persistent",
            default=True,
            view=types.CheckboxView(),
        )

        file_explorer = types.FileExplorerView(
            choose_dir=True,
            button_label="Choose a directory...",
        )
        inputs.file(
            "directory",
            required=True,
            label="Directory",
            description="The directory containing the documents to create a dataset from",
            view=file_explorer,
        )
        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        dataset_name = ctx.params.get("dataset_name", None)
        persistent = ctx.params.get("persistent", True)

        directory = ctx.params["directory"]["absolute_path"]
        documents = SimpleDirectoryReader(input_dir=directory).load_data()
        dataset = create_dataset_from_documents(
            documents, name=dataset_name, persistent=persistent
        )
        ctx.trigger("open_dataset", dict(dataset=dataset.name))


class AddLlamaDocumentsToDataset(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="add_llama_documents_to_dataset",
            label="Add LlamaIndex Documents to Dataset",
            description=(
                "Select a directory containing mixed media"
                "files. These will be converted into LlamaIndex"
                "`Document` and `ImageDocument` instances, and"
                "added to this dataset."
            ),
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        if not ctx.dataset and not ctx.view:
            inputs.view(
                "no_dataset_warning",
                types.Warning(
                    label="No dataset",
                    description="Need dataset to add documents",
                ),
            )
            return types.Property(inputs)

        form_view = types.View(
            label="Add Documents to Dataset",
            description="Add documents to a dataset",
        )

        file_explorer = types.FileExplorerView(
            choose_dir=True,
            button_label="Choose a directory...",
        )
        inputs.file(
            "directory",
            required=True,
            label="Directory",
            description="The directory containing the documents to add",
            view=file_explorer,
        )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        directory = ctx.params["directory"]["absolute_path"]
        documents = SimpleDirectoryReader(directory).load_data()
        add_documents_to_dataset(documents, ctx.dataset)
        ctx.trigger("reload_dataset")


####################################################################################################
####################################################################################################


def _get_string_fields(dataset):
    return list(dataset.get_field_schema(ftype=fo.StringField).keys())


def _get_viable_caption_fields(dataset):
    string_fields = _get_string_fields(dataset)
    return [s for s in string_fields if "llama__" not in s and s != "filepath"]


def _indexing_options_input(ctx, inputs):
    inputs.bool(
        "embed_images",
        label="Embed images with CLIP?",
        description="Uncheck if you want to use captions instead",
        default=True,
        view=types.CheckboxView(),
    )

    if ctx.params.get("embed_images", True):
        return

    viable_caption_fields = _get_viable_caption_fields(ctx.dataset)
    if not viable_caption_fields:
        inputs.view(
            "no_viable_description_fields_warning",
            types.Warning(
                label="No image captions found",
                description=(
                    "To add captions, try out the image-captioning plugin"
                    "https://github.com/jacobmarks/fiftyone-image-captioning-plugin"
                ),
            ),
        )
        inputs.str("caption_field", required=True, view=types.HiddenView())
    else:
        caption_fields_radio_group = types.RadioGroup()
        for caption_field in viable_caption_fields:
            caption_fields_radio_group.add_choice(
                caption_field, label=caption_field
            )
        inputs.enum(
            "caption_field",
            caption_fields_radio_group.values(),
            label="Caption field",
            description="The field to use as the caption for each image",
            view=types.AutocompleteView(),
            required=True,
        )


# def _get_db_path(hash):
#     path = os.path.expanduser(f"~/.fiftyone/llama_index/qdrant_db/{hash}")
#     if not os.path.exists(path):
#         os.makedirs(path)
#     return path


def _create_index(ctx):
    index_hash = _create_hash()

    # client = qdrant_client.QdrantClient(location=":memory:")
    index_name = ctx.params.get("index_name", None)
    dataset = ctx.dataset

    is_image_to_text = not ctx.params.get("embed_images", True)
    if is_image_to_text:
        captions_field = ctx.params.get("caption_field", None)
    else:
        captions_field = None

    ### Gather documents
    text_documents = [
        _llama_doc_from_fo_sample(sample)
        for sample in dataset.match_tags("text")
    ]
    image_documents = [
        _llama_doc_from_fo_sample(sample, text_field=captions_field)
        for sample in dataset.match_tags("image")
    ]
    documents = text_documents + image_documents
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(documents)

    ### Create storage context
    text_collection_name = f"text_collection_{index_hash}"
    text_store = MilvusVectorStore(
        collection_name=text_collection_name, uri=URI, token=TOKEN, dim=1536
    )
    image_collection_name = f"image_collection_{index_hash}"
    image_store = MilvusVectorStore(
        collection_name=image_collection_name, uri=URI, token=TOKEN, dim=512
    )
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )

    ### Create index
    index = MultiModalVectorStoreIndex(
        nodes=nodes,
        is_image_to_text=is_image_to_text,
        storage_context=storage_context,
    )

    cache = get_cache()
    cache_key = f"{dataset.name}_{index_name}"
    if cache_key not in cache:
        cache[cache_key] = index

    ### save it with custom runs
    run_key = f"llamaindex_index_{index_hash}"
    config = dataset.init_run()
    config.name = index_name

    ### serialize and save
    config.nodes = [node for node in nodes]
    config.text_collection_name = text_collection_name
    config.image_collection_name = image_collection_name
    # config.storage_context_for_text = index.storage_context.vector_stores[
    #     "default"
    # ].to_dict()
    # config.storage_context_for_images = index.storage_context.vector_stores[
    #     "image"
    # ].to_dict()
    # config.service_context = index.service_context.to_dict()
    config.is_image_to_text = is_image_to_text
    config.captions_field = captions_field
    dataset.register_run(run_key, config)


class CreateMultiModalRAGIndex(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="create_multimodal_rag_index",
            label="Create MultiModal RAG Index",
            description="Create a multimodal RAG index using LlamaIndex and Milvus",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        if not ctx.dataset or not ctx.view:
            inputs.view(
                "no_dataset_warning",
                types.Warning(
                    label="No dataset",
                    description="Need dataset to create an index",
                ),
            )
            return types.Property(inputs)

        form_view = types.View(
            label="Create MultiModal RAG Index",
            description="Create a multimodal RAG index using LlamaIndex and Milvus",
        )

        inputs.str(
            "index_name",
            label="Index name",
            description="The name of the multimodal RAG index to create",
            required=True,
        )

        _indexing_options_input(ctx, inputs)

        return types.Property(inputs, view=form_view)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        _create_index(ctx)
        ctx.trigger("reload_dataset")


def _get_index_name(ctx, run_key):
    return ctx.dataset.get_run_info(run_key).config.name


def _list_indexes(ctx):
    dataset = ctx.dataset
    run_keys = dataset.list_runs()
    return [
        _get_index_name(ctx, run_key)
        for run_key in run_keys
        if "llamaindex_index" in run_key
    ]


def _get_key_from_name(dataset, index_name):
    run_keys = dataset.list_runs()
    for run_key in run_keys:
        if "llamaindex_index" in run_key:
            run_info = dataset.get_run_info(run_key)
            if run_info.config.name == index_name:
                return run_key
    return None


def _execute_run_info(ctx, run_key):
    info = ctx.dataset.get_run_info(run_key)

    timestamp = info.timestamp.strftime("%Y-%M-%d %H:%M:%S")
    version = info.version

    ## initialize
    odict = {
        "timestamp": timestamp,
        "version": version,
    }

    config = info.config.serialize()
    config = {k: v for k, v in config.items() if v is not None}
    is_image_to_text = config.get("is_image_to_text", False)

    if is_image_to_text:
        caption_field = config.get("captions_field", "")
        odict["image_to_text"] = "using text from field: " + caption_field
    else:
        odict["image_to_text"] = "using CLIP embeddings"

    return odict


def _initialize_run_output(ctx):
    outputs = types.Object()
    outputs.str("timestamp", label="Creation time")
    outputs.str("version", label="FiftyOne version")
    outputs.bool("image_to_text", label="Image <> Text Comparison")
    return outputs


class GetMultimodalRAGIndexInfo(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_multimodal_rag_index_info",
            label="Get info about a multimodal RAG index",
            icon="/assets/icon.svg",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        if not ctx.dataset or not ctx.view:
            inputs.view(
                "no_dataset_warning",
                types.Warning(
                    label="No dataset", description="Need dataset to query"
                ),
            )
            return types.Property(inputs)
        indexes = _list_indexes(ctx)
        if not indexes:
            inputs.view(
                "no_indexes_warning",
                types.Warning(
                    label="No multimodal RAG indexes", description="No indexes"
                ),
            )
            return types.Property(inputs)

        indexes_radio_group = types.RadioGroup()
        for index in indexes:
            indexes_radio_group.add_choice(index, label=index)
        inputs.enum(
            "index_name",
            indexes_radio_group.values(),
            label="Index name",
            description="The name of the multimodal RAG index to inspect",
            view=types.AutocompleteView(),
            required=True,
        )
        return types.Property(inputs)

    def execute(self, ctx):
        index_name = ctx.params.get("index_name", None)
        run_key = _get_key_from_name(ctx.dataset, index_name)
        return _execute_run_info(ctx, run_key)

    def resolve_output(self, ctx):
        outputs = _initialize_run_output(ctx)
        view = types.View(label="Multimodal RAG Index info")
        return types.Property(outputs, view=view)


def _model_choices_input(ctx, inputs):
    model_choices = ALL_MODEL_NAMES

    models_radio_group = types.RadioGroup()

    for model_choice in model_choices:
        models_radio_group.add_choice(model_choice, label=model_choice)

    inputs.enum(
        "model_name",
        models_radio_group.values(),
        label="MultiModal LLM",
        description="The multimodal language model to use for the query",
        view=types.AutocompleteView(),
        required=True,
        default="gpt-4-vision-preview",
    )


def _top_k_input(ctx, inputs):
    inputs.int(
        "text_top_k",
        label="Top k text results",
        description="The number of top k text results to return",
        default=2,
        required=True,
    )
    inputs.int(
        "image_top_k",
        label="Top k image results",
        description="The number of top k image results to return",
        default=2,
        required=True,
    )


def _get_multimodal_llm(llm_name, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    if llm_name == "gpt-4-vision-preview":
        constructor, _model_name = OpenAIMultiModal, llm_name
    elif "gemini" in llm_name:
        constructor, _model_name = GeminiMultiModal, "models/" + llm_name
    elif llm_name in REPLICATE_MODEL_NAMES:
        constructor, _model_name = (
            ReplicateMultiModal,
            REPLICATE_MULTI_MODAL_LLM_MODELS[llm_name],
        )

    return constructor(model=_model_name, max_new_tokens=max_new_tokens)


def _load_index(dataset, run_key):
    config = dataset.get_run_info(run_key).config
    index_name = config.name

    cache = get_cache()
    cache_key = f"{dataset.name}_{index_name}"
    if cache_key in cache:
        return cache[cache_key]

    nodes = []
    for node in config.nodes:
        if "image" in node["metadata"]["file_type"]:
            nodes.append(ImageNode.from_dict(node))
        else:
            nodes.append(TextNode.from_dict(node))

    is_image_to_text = config.is_image_to_text

    text_storage_context = MilvusVectorStore(
        collection_name=config.text_collection_name,
        uri=URI,
        token=TOKEN,
        dim=1536,
    )

    image_storage_context = MilvusVectorStore(
        collection_name=config.image_collection_name,
        uri=URI,
        token=TOKEN,
        dim=512,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=text_storage_context, image_store=image_storage_context
    )
    ### recreate index
    index = MultiModalVectorStoreIndex(
        nodes=nodes,
        is_image_to_text=is_image_to_text,
        storage_context=storage_context,
    )

    cache[cache_key] = index
    return index


def _query_index(ctx):
    dataset = ctx.dataset
    query = ctx.params.get("query", None)
    index_name = ctx.params.get("index_name", None)
    run_key = _get_key_from_name(ctx.dataset, index_name)
    index = _load_index(dataset, run_key)

    model_name = ctx.params.get("model_name", "gpt-4-vision-preview")
    model = _get_multimodal_llm(model_name)
    from llama_index.core.multi_modal_llms.base import MultiModalLLM

    text_top_k = ctx.params.get("text_top_k", 2)
    image_top_k = ctx.params.get("image_top_k", 2)

    query_engine = index.as_query_engine(llm=model, text_qa_template=qa_tmpl)

    query_engine.retriever.similarity_top_k = text_top_k
    if "llava" in model_name:
        query_engine.retriever.image_similarity_top_k = 1
    else:
        query_engine.retriever.image_similarity_top_k = image_top_k

    response = query_engine.query(query)
    nodes = response.metadata["text_nodes"] + response.metadata["image_nodes"]
    sample_ids, scores = zip(
        *[(n.metadata["fiftyone_sample_id"], n.score) for n in nodes]
    )
    view = dataset.select(sample_ids, ordered=True)
    view.set_values("score", scores)
    # strategy = resolve_strategy(ctx)...  # TODO

    output = {
        "response": response.response,
        "index_name": index_name,
        "model_name": model_name,
    }

    return view, output


class QueryMultiModalRAGIndex(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="query_multimodal_rag_index",
            label="Query MultiModal RAG Index",
            description="Query a multimodal RAG index using LlamaIndex and Milvus",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        if not ctx.dataset or not ctx.view:
            inputs.view(
                "no_dataset_warning",
                types.Warning(
                    label="No dataset", description="Need dataset to query"
                ),
            )
            return types.Property(inputs)

        indexes = _list_indexes(ctx)
        if not indexes:
            inputs.view(
                "no_indexes_warning",
                types.Warning(
                    label="No multimodal RAG indexes", description="No indexes"
                ),
            )
            return types.Property(inputs)

        inputs.str(
            "query",
            label="Query",
            description="The query to use to query the multimodal RAG index",
            required=True,
        )

        indexes_radio_group = types.RadioGroup()
        for index in indexes:
            indexes_radio_group.add_choice(index, label=index)
        inputs.enum(
            "index_name",
            indexes_radio_group.values(),
            label="Index name",
            description="The name of the multimodal RAG index to query",
            view=types.AutocompleteView(),
            required=True,
        )

        _model_choices_input(ctx, inputs)
        _top_k_input(ctx, inputs)
        ### choose strategy...

        return types.Property(inputs)

    def execute(self, ctx):
        view, output = _query_index(ctx)
        ctx.trigger(
            "set_view",
            params=dict(view=serialize_view(view)),
        )
        return output

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("index_name", label="Index name")
        outputs.str("model_name", label="Model name")
        outputs.str(
            "response",
            label="Response",
            view=types.MarkdownView(read_only=True),
        )
        view = types.View(label="Retrieval Augmented Generation:")
        return types.Property(outputs, view=view)


def register(plugin):
    plugin.register(CreateDatasetFromLlamaDocuments)
    plugin.register(AddLlamaDocumentsToDataset)
    plugin.register(CreateMultiModalRAGIndex)
    plugin.register(QueryMultiModalRAGIndex)
    plugin.register(GetMultimodalRAGIndexInfo)

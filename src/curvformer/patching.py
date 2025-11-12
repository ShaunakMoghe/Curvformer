# src/curvformer/patching.py
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertLayer
from .curv_attention import CurvSelfAttention

def patch_bert_self_attention(model: BertModel, tau: float = 16.0, layers: str = "all"):
    """Replace BertSelfAttention with CurvSelfAttention in-place.
    layers: "all" or comma-separated indices, e.g., "0,1,2".
    """
    if layers == "all":
        idxs = range(len(model.bert.encoder.layer))
    else:
        idxs = [int(x) for x in layers.split(",")]

    for i in idxs:
        layer: BertLayer = model.bert.encoder.layer[i]
        curv_attn = CurvSelfAttention(model.config, tau=tau)
        # copy query/key/value weights from existing attention for a fair start
        curv_attn.query.load_state_dict(layer.attention.self.query.state_dict())
        curv_attn.key.load_state_dict(layer.attention.self.key.state_dict())
        curv_attn.value.load_state_dict(layer.attention.self.value.state_dict())
        layer.attention.self = curv_attn
    return model
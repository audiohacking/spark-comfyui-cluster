"""spark-comfyui-cluster: ComfyUI-Continuity on the dgx-monarch cluster.

Baked into the cluster image beside dgx-monarch (/opt/monarch-nodes), so it
loads only where the cluster nodes do. It gives Continuity's MiniMax H3 family
a "monarch" sampling backend; its two nodes exist only inside Continuity's
expanded graph. See continuity_monarch.py for what that means and what it
refuses.

ComfyUI loads the image's custom-node roots after its own custom_nodes, so
Continuity is normally imported already. Whatever is missing at load (Continuity,
or dgx-monarch when it loads after this pack) is installed by the next prompt:
expansion happens at execution, after every on_prompt handler has run.
"""

import logging

from . import continuity_monarch

from . import face_route
from . import monarch_preview
from . import te_cache
from . import vae_cache
from . import output_formats
from . import mux_threads

te_cache.install()  # every workflow's text encodes, Continuity or not
vae_cache.install()  # every workflow's VAE encodes, Continuity or not
output_formats.install()  # Continuity's saved files show up in the queue and assets

# Built only inside Continuity's expanded graph (continuity_monarch.py).
NODE_CLASS_MAPPINGS = {monarch_preview.NODE_ID: monarch_preview.SparkMonarchPreview,
                       face_route.NODE_ID: face_route.SparkMonarchFaceRoute}
NODE_DISPLAY_NAME_MAPPINGS = {monarch_preview.NODE_ID: "Cluster preview (spark)",
                              face_route.NODE_ID: "Cluster face pass route (spark)"}

try:
    import server
    from aiohttp import web

    @server.PromptServer.instance.routes.get("/spark/continuity-monarch")
    async def _status(request):
        return web.json_response({**continuity_monarch.status(),
                                  "te_cache": te_cache.status(),
                                  "vae_cache": vae_cache.status()})
except Exception:
    logging.getLogger("spark.continuity_monarch").exception(
        "Continuity cluster backend: no status route")

def _settle():
    """Install whatever is not in place yet. ComfyUI loads the folders under
    /opt/monarch-nodes in directory order, which is not guaranteed, so this
    pack can load before dgx-monarch: its sampler is then not registered and
    its package not importable yet. Each step is idempotent."""
    monarch_preview.quiet_notices()
    vae_cache.install()  # back on top if dgx-monarch's encode split wrapped it since
    mux_threads.install()  # Continuity may have loaded after this pack
    installed = continuity_monarch.install()
    if installed:
        monarch_preview.wrap_sampler()
    return installed


_settle()

try:
    import server

    def _on_prompt(json_data):
        try:
            _settle()
        except Exception:  # never fail a prompt over the overlay
            logging.getLogger("spark.continuity_monarch").exception(
                "Continuity cluster backend: install failed")
        return json_data

    server.PromptServer.instance.add_on_prompt_handler(_on_prompt)
except Exception:
    logging.getLogger("spark.continuity_monarch").warning(
        "Continuity cluster backend: no prompt hook to finish installing from")

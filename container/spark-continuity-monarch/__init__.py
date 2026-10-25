"""spark-comfyui-cluster: ComfyUI-Continuity on the dgx-monarch cluster.

Baked into the cluster image beside dgx-monarch (/opt/monarch-nodes), so it
loads only where the cluster nodes do. Adds no nodes of its own; it gives
Continuity's MiniMax H3 family a "monarch" sampling backend. See
continuity_monarch.py for what that means and what it refuses.

ComfyUI loads the image's custom-node roots after its own custom_nodes, so
Continuity is normally imported already. If not, the first prompt installs it:
expansion happens at execution, after every on_prompt handler has run.
"""

import logging

from . import continuity_monarch

from . import monarch_preview

# Built only inside Continuity's expanded graph (continuity_monarch.py).
NODE_CLASS_MAPPINGS = {monarch_preview.NODE_ID: monarch_preview.SparkMonarchPreview}
NODE_DISPLAY_NAME_MAPPINGS = {monarch_preview.NODE_ID: "Cluster preview (spark)"}

try:
    import server
    from aiohttp import web

    @server.PromptServer.instance.routes.get("/spark/continuity-monarch")
    async def _status(request):
        return web.json_response(continuity_monarch.status())
except Exception:
    logging.getLogger("spark.continuity_monarch").exception(
        "Continuity cluster backend: no status route")

monarch_preview.quiet_notices()

if not continuity_monarch.install():
    try:
        import server

        def _on_prompt(json_data):
            try:
                continuity_monarch.install()
            except Exception:  # never fail a prompt over the overlay
                logging.getLogger("spark.continuity_monarch").exception(
                    "Continuity cluster backend: install failed")
            return json_data

        server.PromptServer.instance.add_on_prompt_handler(_on_prompt)
    except Exception:
        logging.getLogger("spark.continuity_monarch").warning(
            "Continuity cluster backend: not installed (no Continuity yet, and "
            "no prompt hook to retry from)")

from avipe_mls import unet

def get_checkpoint_versions(model_manager:unet.models._ModelManager | None) -> list[tuple[str,str]]:
    #TODO: make this a tuple of (friendly name, filename)
    versions = []
    
    if not model_manager:
        return versions

    for name in model_manager.checkpoints:
                versions.append((name, name))
    if model_manager.latest:
        versions.append(("latest", "latest"))
    return versions
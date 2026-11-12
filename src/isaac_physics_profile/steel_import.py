"""Import a locally exported scene in the Newton wheel's OpenUSD process."""

import pickle
import sys
from pathlib import Path

from newton import ModelBuilder
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx
from ovphysx.samples.python_samples.schema_support import physx_schema_paths
from pxr import Plug, Usd


def main():
    for schema in physx_schema_paths():
        Plug.Registry().RegisterPlugins(str(schema))
    stage = Usd.Stage.Open(sys.argv[1])
    builder = ModelBuilder()
    expected = set(builder.custom_attributes)
    if sys.argv[3] != "default":
        builder.rigid_gap = float(sys.argv[3])
    worlds = sys.argv[4:]
    options = dict(
        load_visual_shapes=False,
        load_static_visual_shapes=False,
        schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
    )
    builder.add_usd(stage, ignore_paths=worlds, **options)
    for root in worlds:
        builder.begin_world()
        builder.add_usd(stage, root_path=root, **options)
        builder.end_world()
    data = vars(builder).copy()
    attrs = data.pop("custom_attributes")
    if attrs.keys() != expected:
        raise ValueError("USD added unrecognized Newton custom attributes")
    values = {name: attr.values for name, attr in attrs.items()}
    Path(sys.argv[2]).write_bytes(pickle.dumps((data, values), protocol=5))


if __name__ == "__main__":
    main()

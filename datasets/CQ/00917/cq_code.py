import cadquery as cq

result = (
cq.Workplane("XY")
.sphere(5)
.transformed(offset=(10, 0, 0))
.sphere(5)
.transformed(offset=(10, 0, 0))
.sphere(5)
.compounds()
)
cq.exporters.export(result, 'GT.stl')
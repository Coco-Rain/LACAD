import cadquery as cq

result = (
cq.Workplane("XY")
.box(5, 5, 1)
.transformed(offset=(7, 0, 0))
.box(5, 5, 1)
.transformed(offset=(7, 0, 0))
.box(5, 5, 1)
.compounds()
)
cq.exporters.export(result, 'GT.stl')
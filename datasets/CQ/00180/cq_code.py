import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 5)
.edges()
.fillet(1)
)
cq.exporters.export(result, 'GT.stl')
import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 10)
.edges("|Z")
.fillet(1)
)
cq.exporters.export(result, 'GT.stl')
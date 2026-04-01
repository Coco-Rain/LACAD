import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 1)
.edges("|Z")
.fillet(0.5)
)
cq.exporters.export(result, 'GT.stl')
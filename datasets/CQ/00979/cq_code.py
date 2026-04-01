import cadquery as cq

result = (
cq.Workplane("XY")
.box(8, 8, 2)
.edges("|Z")
.first()
.chamfer(1)
)
cq.exporters.export(result, 'GT.stl')
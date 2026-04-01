import cadquery as cq

result = (
cq.Workplane("XY")
.rect(20, 10)
.extrude(5)
.edges("|Z")
.chamfer(1)
)
cq.exporters.export(result, 'GT.stl')
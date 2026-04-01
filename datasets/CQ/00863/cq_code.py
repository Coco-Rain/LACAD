import cadquery as cq

result = (
cq.Workplane("XY")
.rect(20, 20)
.extrude(5)
.edges("|Z")
.chamfer(2)
)
cq.exporters.export(result, 'GT.stl')
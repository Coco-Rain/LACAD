import cadquery as cq

result = (
cq.Workplane("XY")
.polygon(6, 10)
.extrude(5)
.edges("|Z")
.chamfer(1)
)
cq.exporters.export(result, 'GT.stl')
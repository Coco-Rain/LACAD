import cadquery as cq

result = (
cq.Workplane("XY")
.rect(20, 10)
.extrude(5)
.edges("|Z")
.fillet(1)
)
cq.exporters.export(result, 'GT.stl')
import cadquery as cq

result = (
cq.Workplane("XY")
.polygon(5, 40)
.extrude(10)
.vertices(">Z")
.circle(2)
.extrude(10)
)
cq.exporters.export(result, 'GT.stl')
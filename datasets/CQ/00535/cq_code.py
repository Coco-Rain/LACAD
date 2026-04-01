import cadquery as cq

result = (
cq.Workplane("XY")
.circle(5)
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')
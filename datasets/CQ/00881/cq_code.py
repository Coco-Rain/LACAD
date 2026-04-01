import cadquery as cq

result = (
cq.Workplane("XY")
.circle(5)
.twistExtrude(20, 180)
)
cq.exporters.export(result, 'GT.stl')
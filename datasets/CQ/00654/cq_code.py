import cadquery as cq

result = (
cq.Workplane("XY")
.rect(5, 5)
.twistExtrude(30, 360)
)
cq.exporters.export(result, 'GT.stl')
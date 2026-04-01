import cadquery as cq

result = (
cq.Workplane("XY")
.center(2, 3)
.rect(4, 6)
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')
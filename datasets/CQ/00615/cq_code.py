import cadquery as cq

result = (
cq.Workplane("XY")
.rect(10, 10)
.offset2D(2)
.extrude(5)
)
cq.exporters.export(result, 'GT.stl')
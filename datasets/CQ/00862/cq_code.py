import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(0, 0)
.lineTo(5, 0)
.lineTo(5, 3)
.lineTo(0, 3)
.close()
.wire()
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')
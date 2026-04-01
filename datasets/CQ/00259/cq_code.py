import cadquery as cq

result = (
cq.Workplane("XY").moveTo(10, 0)
.vLine(22)
.lineTo(25, 42)
.threePointArc((0, 53), (-25, 42))
.lineTo(-10, 22)
.vLine(-22)
.close()
.extrude(2.0)
)
cq.exporters.export(result, 'GT.stl')
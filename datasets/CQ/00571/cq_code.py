import cadquery as cq

result = (
cq.Workplane("front")
.spline([(0, 0), (3, 5), (6, 0)])
.lineTo(0, 0)
.close()
.extrude(4)
)
cq.exporters.export(result, 'GT.stl')
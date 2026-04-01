import cadquery as cq

result = (
cq.Workplane("XZ")
.moveTo(0, 0)
.lineTo(0, 20)
.spline([(0, 20), (5, 25), (10, 20)])
.lineTo(10, 0)
.close()
.revolve(360)
)
cq.exporters.export(result, 'GT.stl')
import cadquery as cq

points = [(2, 2), (5, 5), (8, 8), (10, 10)]
result = (cq.Sketch()
.push(points)
.circle(1)
)
cq.exporters.export(result, 'GT.stl')
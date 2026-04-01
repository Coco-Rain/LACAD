import cadquery as cq

points = [(2, 2), (5, 5), (8, 8), (10, 10)]
result = (cq.Sketch()
.push(points)
.rect(1, 1)
.rect(2, 2)
.rect(3, 3)
)
cq.exporters.export(result, 'GT.stl')
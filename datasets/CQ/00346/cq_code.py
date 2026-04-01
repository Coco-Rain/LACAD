import cadquery as cq

path = cq.Edge.makeLine((50, 15, 50), (35, 25, 10))
result = (
cq.Workplane("XY")
.ellipse(1, 2)
.sweep(path)
)
cq.exporters.export(result, 'GT.stl')
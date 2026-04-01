import cadquery as cq

points = [(-10, 0), (10, 0), (30, 0)]
result = (
cq.Workplane("XZ")
.moveTo(10, 0)
.box(60, 13, 24, centered=(True, True, False))
.faces("<Y")
.workplane(centerOption='ProjectedOrigin', origin=(0, 0, 0))
.pushPoints(points)
.cboreHole(6, 10, 4)
)
cq.exporters.export(result, 'GT.stl')
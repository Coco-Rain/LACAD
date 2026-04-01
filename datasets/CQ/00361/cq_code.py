import cadquery as cq

result = (
cq.Workplane("front")
.box(3, 4, 0.25)
.pushPoints([(0, 0.75), (0, -0.75)])
.polygon(6, 1)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')
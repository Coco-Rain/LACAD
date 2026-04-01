import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 5)
.faces("|Z")
.first()
.workplane()
.center(0, 2)
.circle(1)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')
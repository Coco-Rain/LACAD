import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 30, 10)
.faces(">Z")
.workplane()
.center(5, -5)
.circle(6)
.cutThruAll()
.rotateAboutCenter((1, 0, 0), 70)
)
cq.exporters.export(result, 'GT.stl')
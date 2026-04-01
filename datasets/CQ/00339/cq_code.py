import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 30, 10)
.faces(">Z")
.workplane()
.center(5, -5)
.rect(10, 6)
.extrude(5)
.rotateAboutCenter((1, 0, 0), 30)
)
cq.exporters.export(result, 'GT.stl')
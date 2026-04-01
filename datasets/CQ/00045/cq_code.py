import cadquery as cq

result = (
cq.Workplane("front")
.box(66, 10, 70)
.faces(">X")
.workplane(centerOption = 'CenterOfBoundBox')
.move(0, -32)
.line(38, 0)
.line(0, 60)
.close()
.extrude(-3)
)
cq.exporters.export(result, 'GT.stl')
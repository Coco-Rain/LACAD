import cadquery as cq

result = (
cq.Workplane("YZ")
.box(3, 10, 6)
.faces(">X")
.workplane()
.move(0, 4)
.move(-2.5, 0)
.rect(1.5, 3)
.extrude(2, both=False)
)
cq.exporters.export(result, 'GT.stl')
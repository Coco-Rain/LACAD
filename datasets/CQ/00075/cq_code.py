import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 5)
.faces(">Z")
.workplane()
.center(10, 10)
.rect(4, 4)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')
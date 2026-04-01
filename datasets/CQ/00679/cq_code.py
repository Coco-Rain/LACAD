import cadquery as cq

result = (
cq.Workplane("YZ")
.box(10, 20, 8)
.faces(">X")
.workplane()
.center(2, 6)
.hole(3, 4)
)
cq.exporters.export(result, 'GT.stl')
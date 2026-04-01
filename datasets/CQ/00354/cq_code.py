import cadquery as cq

result = (
cq.Workplane("YZ")
.box(10, 20, 30)
.faces(">Y")
.workplane()
.rect(5, 5)
.extrude(10)
)
cq.exporters.export(result, 'GT.stl')
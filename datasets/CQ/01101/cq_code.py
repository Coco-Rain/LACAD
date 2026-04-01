import cadquery as cq

result = (
cq.Workplane("YZ")
.box(5, 5, 1)
.faces("<X")
.workplane()
.center(-1, -2)
.rect(2, 3)
.extrude(-1)
cq.exporters.export(result, 'GT.stl')
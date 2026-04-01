import cadquery as cq

result = (
cq.Workplane("XY")
.center(2.5, 2.5)
.rect(5, 5)
.extrude(1)
.faces(">Z")
.workplane()
.center(-1.25, -1.25)
.rect(2.5, 2.5)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')
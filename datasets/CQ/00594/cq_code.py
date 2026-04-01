import cadquery as cq

result = (
cq.Workplane("XY")
.rect(5, 10)
.extrude(2)
.faces(">Z")
.workplane()
.circle(1.5)
.cutThruAll()
.mirror("XZ")
)
cq.exporters.export(result, 'GT.stl')
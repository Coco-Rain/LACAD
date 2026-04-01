import cadquery as cq

result = (
cq.Workplane("XY")
.rect(30, 20).extrude(8)
.faces(">Z").workplane()
.ellipse(12, 8).extrude(6)
.faces(">Z").shell(1.5)
)
cq.exporters.export(result, 'GT.stl')
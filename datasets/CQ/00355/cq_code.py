import cadquery as cq

result = (
cq.Workplane("XY")
.rect(30, 20).extrude(5)
.faces(">Z").workplane()
.ellipse(12, 8).extrude(3)
.faces(">Z").workplane()
.slot2D(15, 4).cutThruAll()
.edges("|Z").fillet(1.5)
)
cq.exporters.export(result, 'GT.stl')
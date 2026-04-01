import cadquery as cq

result = (
cq.Workplane("XY")
.circle(10).extrude(5)
.faces(">Z").workplane()
.polygon(6, 15).extrude(3)
.edges("|Z").fillet(2)
.union(
cq.Workplane("XY").transformed(offset=(20,0,0))
.circle(5).extrude(8)
.edges(">Z").fillet(1.5)
)
)
result
cq.exporters.export(result, 'GT.stl')
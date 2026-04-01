import cadquery as cq

result = (
cq.Workplane("XY")
.circle(10).extrude(5)
.faces(">Z")
.polygon(6, 8).extrude(4)
.faces(">Z")
.workplane()
.cskHole(3.5, 7.0, 2, depth=6)
.edges("|Z").fillet(1.5)
)
cq.exporters.export(result, 'GT.stl')
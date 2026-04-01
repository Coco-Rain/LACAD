import cadquery as cq

result = (
cq.Workplane("XY")
.polygon(6, 18)
.extrude(12)
.faces(">Z")
.workplane()
.cylinder(8, 7)
.edges("|Z")
.chamfer(2.5)
)
cq.exporters.export(result, 'GT.stl')